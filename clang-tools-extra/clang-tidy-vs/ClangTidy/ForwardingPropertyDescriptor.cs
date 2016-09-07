using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{
    /// <summary>
    /// A decorator of sorts.  Accepts a PropertyDescriptor to its constructor
    /// and forwards all calls to the underlying PropertyDescriptor.  In this way
    /// we can inherit from ForwardingPropertyDescriptor and override only the
    /// few methods we need to customize the behavior of, while allowing the
    /// underlying PropertyDescriptor to do the real work.
    /// </summary>
    public abstract class ForwardingPropertyDescriptor : PropertyDescriptor
    {
        private readonly PropertyDescriptor root;
        protected PropertyDescriptor Root { get { return root; } }
        protected ForwardingPropertyDescriptor(PropertyDescriptor root)
            : base(root)
        {
            this.root = root;
        }

        public override void AddValueChanged(object component, EventHandler handler)
        {
            root.AddValueChanged(component, handler);
        }

        public override AttributeCollection Attributes
        {
            get
            {
                return root.Attributes;
            }
        }

        public override bool CanResetValue(object component)
        {
            return root.CanResetValue(component);
        }

        public override string Category
        {
            get
            {
                return root.Category;
            }
        }

        public override Type ComponentType
        {
            get
            {
                return root.ComponentType;
            }
        }

        public override TypeConverter Converter
        {
            get
            {
                return root.Converter;
            }
        }

        public override string Description
        {
            get
            {
                return root.Description;
            }
        }

        public override bool DesignTimeOnly
        {
            get
            {
                return root.DesignTimeOnly;
            }
        }

        public override string DisplayName
        {
            get
            {
                return root.DisplayName;
            }
        }

        public override bool Equals(object obj)
        {
            return root.Equals(obj);
        }

        public override PropertyDescriptorCollection GetChildProperties(object instance, Attribute[] filter)
        {
            return root.GetChildProperties(instance, filter);
        }

        public override object GetEditor(Type editorBaseType)
        {
            return root.GetEditor(editorBaseType);
        }

        public override int GetHashCode()
        {
            return root.GetHashCode();
        }

        public override object GetValue(object component)
        {
            return root.GetValue(component);
        }

        public override bool IsBrowsable
        {
            get
            {
                return root.IsBrowsable;
            }
        }

        public override bool IsLocalizable
        {
            get
            {
                return root.IsLocalizable;
            }
        }

        public override bool IsReadOnly
        {
            get
            {
                return root.IsReadOnly;
            }
        }

        public override string Name
        {
            get
            {
                return root.Name;
            }
        }

        public override Type PropertyType
        {
            get
            {
                return root.PropertyType;
            }
        }

        public override void RemoveValueChanged(object component, EventHandler handler)
        {
            root.RemoveValueChanged(component, handler);
        }

        public override void ResetValue(object component)
        {
            root.ResetValue(component);
        }

        public override void SetValue(object component, object value)
        {
            root.SetValue(component, value);
        }

        public override bool ShouldSerializeValue(object component)
        {
            return root.ShouldSerializeValue(component);
        }

        public override bool SupportsChangeEvents
        {
            get
            {
                return root.SupportsChangeEvents;
            }
        }

        public override string ToString()
        {
            return root.ToString();
        }
    }
}
