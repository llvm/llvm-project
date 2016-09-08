using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{
    /// <summary>
    /// The goal of this class is to enable displaying of a PropertyGrid in much the
    /// same way that Visual Studio's C++ project system does.  A project or file can
    /// have properties which might inherit from their parent, or be overridden.
    /// It turns out this is somewhat non-trivial.  The .NET PropertyGrid is good makes
    /// displaying simple properties with a static notion of what constitutes a
    /// "default" value very easy.  You simply apply an Attribute to the class that says
    /// what the default value is and you're done.  But when you try to introduce the idea
    /// that a property's default value depends on some other factor, things get much more
    /// complicated due to the static nature of Attributes.
    /// 
    /// The solution to this is to inherit from ICustomTypeDescriptor.  This is the mechanism
    /// by which you can inject or modify attributes or properties at runtime.  The .NET
    /// PropertyGrid is designed in such a way that instead of using simple .NET Reflection to
    /// look for the properties and attributes on a class, it will invoke the methods of
    /// ICustomTypeDescriptor (if your type inherits from it), and ask those methods.  Our
    /// implementation of ICustomTypeDescriptor works by waiting until the PropertyGrid requests
    /// PropertyDescriptors for each of the properties, and then "decorating" them with our
    /// own custom PropertyDescriptor implementation which understands the proeprty inheritance
    /// model we wish to implement.
    /// </summary>
    public partial class DynamicPropertyComponent : Component, ICustomTypeDescriptor
    {
        PropertyDescriptorCollection DynamicProperties_ = new PropertyDescriptorCollection(null);
        private DynamicPropertyComponent Parent_;

        public DynamicPropertyComponent(DynamicPropertyComponent Parent)
        {
            Parent_ = Parent;
        }

        public DynamicPropertyComponent(DynamicPropertyComponent Parent, IContainer container)
        {
            Parent_ = Parent;

            container.Add(this);
            InitializeComponent();
        }

        public AttributeCollection GetAttributes()
        {
            return TypeDescriptor.GetAttributes(GetType());
        }

        public string GetClassName()
        {
            return TypeDescriptor.GetClassName(GetType());
        }

        public string GetComponentName()
        {
            return TypeDescriptor.GetComponentName(GetType());
        }

        public TypeConverter GetConverter()
        {
            return TypeDescriptor.GetConverter(GetType());
        }

        public EventDescriptor GetDefaultEvent()
        {
            return TypeDescriptor.GetDefaultEvent(GetType());
        }

        public PropertyDescriptor GetDefaultProperty()
        {
            return TypeDescriptor.GetDefaultProperty(GetType());
        }

        public object GetEditor(Type editorBaseType)
        {
            return TypeDescriptor.GetEditor(GetType(), editorBaseType);
        }

        public EventDescriptorCollection GetEvents()
        {
            return TypeDescriptor.GetEvents(GetType());
        }

        public EventDescriptorCollection GetEvents(Attribute[] attributes)
        {
            return TypeDescriptor.GetEvents(GetType(), attributes);
        }

        public PropertyDescriptorCollection GetProperties()
        {
            return DynamicProperties_;
        }

        public PropertyDescriptorCollection GetProperties(Attribute[] attributes)
        {
            var Props = DynamicProperties_.OfType<PropertyDescriptor>();
            var Filtered = Props.Where(x => x.Attributes.Contains(attributes)).ToArray();
            return new PropertyDescriptorCollection(Filtered);
        }

        public object GetPropertyOwner(PropertyDescriptor pd)
        {
            return this;
        }

        public void SetDynamicValue<T>(string Name, T Value)
        {
            Name = Name.Replace('-', '_');
            DynamicPropertyDescriptor<T> Descriptor = (DynamicPropertyDescriptor<T>)DynamicProperties_.Find(Name, false);
            Descriptor.SetValue(this, Value);
        }

        public T GetDynamicValue<T>(string Name)
        {
            Name = Name.Replace('-', '_');
            DynamicPropertyDescriptor<T> Descriptor = (DynamicPropertyDescriptor<T>)DynamicProperties_.Find(Name, false);
            return (T)Descriptor.GetValue(this);
        }

        protected void AddDynamicProperty<T>(string Name, Attribute[] Attributes)
        {
            Name = Name.Replace('-', '_');

            // If we have a parent, find the corresponding PropertyDescriptor with the same
            // name from the parent.
            DynamicPropertyDescriptor<T> ParentDescriptor = null;
            if (Parent_ != null)
                ParentDescriptor = (DynamicPropertyDescriptor<T>)Parent_.GetProperties().Find(Name, false);

            DynamicProperties_.Add(new DynamicPropertyDescriptor<T>(Name, ParentDescriptor, Name, Attributes));
        }
    }
}
