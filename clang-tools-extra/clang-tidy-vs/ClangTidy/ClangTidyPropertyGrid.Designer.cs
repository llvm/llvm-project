namespace LLVM.ClangTidy
{
    partial class ClangTidyPropertyGrid
    {
        /// <summary> 
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.label1 = new System.Windows.Forms.Label();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.button1 = new System.Windows.Forms.Button();
            this.propertyGrid1 = new System.Windows.Forms.PropertyGrid();
            this.clangTidyProperties1 = new LLVM.ClangTidy.ClangTidyProperties();
            this.clangTidyConfigurationPage1 = new LLVM.ClangTidy.ClangTidyConfigurationPage();
            this.linkLabelPath = new System.Windows.Forms.LinkLabel();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(14, 17);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(88, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Configuration File";
            // 
            // textBox1
            // 
            this.textBox1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBox1.Location = new System.Drawing.Point(108, 14);
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(222, 20);
            this.textBox1.TabIndex = 1;
            // 
            // button1
            // 
            this.button1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.button1.Location = new System.Drawing.Point(336, 14);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(78, 20);
            this.button1.TabIndex = 2;
            this.button1.Text = "Browse";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // propertyGrid1
            // 
            this.propertyGrid1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.propertyGrid1.Location = new System.Drawing.Point(20, 73);
            this.propertyGrid1.Name = "propertyGrid1";
            this.propertyGrid1.SelectedObject = this.clangTidyProperties1;
            this.propertyGrid1.Size = new System.Drawing.Size(391, 384);
            this.propertyGrid1.TabIndex = 6;
            this.propertyGrid1.ViewBorderColor = System.Drawing.SystemColors.ControlDarkDark;
            this.propertyGrid1.PropertyValueChanged += new System.Windows.Forms.PropertyValueChangedEventHandler(this.propertyGrid1_PropertyValueChanged);
            // 
            // linkLabelPath
            // 
            this.linkLabelPath.AutoSize = true;
            this.linkLabelPath.Location = new System.Drawing.Point(29, 50);
            this.linkLabelPath.Name = "linkLabelPath";
            this.linkLabelPath.Size = new System.Drawing.Size(55, 13);
            this.linkLabelPath.TabIndex = 7;
            this.linkLabelPath.TabStop = true;
            this.linkLabelPath.Text = "linkLabel1";
            this.linkLabelPath.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabelPath_LinkClicked);
            // 
            // ClangTidyPropertyGrid
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.linkLabelPath);
            this.Controls.Add(this.propertyGrid1);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.label1);
            this.Name = "ClangTidyPropertyGrid";
            this.Size = new System.Drawing.Size(444, 469);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.PropertyGrid propertyGrid1;
        private ClangTidyProperties clangTidyProperties1;
        private ClangTidyConfigurationPage clangTidyConfigurationPage1;
        private System.Windows.Forms.LinkLabel linkLabelPath;
    }
}
